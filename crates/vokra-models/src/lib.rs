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
// vokra_ops::beam_search for the attention-decoder search) rather than
// duplicating.
// SoTA plan Phase X (2026-07-25): forced-alignment ops
// (CLAUDE.md 音声特化オペレータ §"Alignment / Duration / Prosody" —
// `force_align`). Two members:
//   * `ctc_segmentation` (Kürzinger et al., Interspeech 2020; reference
//     implementation github.com/lumaku/ctc-segmentation, Apache-2.0) is a
//     pure host-side algorithm — Viterbi over the standard CTC extended
//     sequence — with no external weights. Emits `Vec<AlignedToken>` for
//     word / sub-word / character granularity uniformly.
//   * `align::charsiu` — Wav2Vec2-based neural forced aligner (real
//     forward, 2026-07-30). Runs the raw-waveform stem
//     (`vokra_ops::waveform_frontend`) → feature projection → n_layer
//     pre-norm Transformer encoder → CTC head → log-softmax →
//     `ctc_segmentation` end-to-end. Real weights bind via
//     `Charsiu::new(CharsiuConfig, CharsiuWeights)`; the real-GGUF
//     binder (`from_gguf`) is a follow-up wave gated on the upstream
//     tensor-name manifest (T29-equivalent). License = MIT (permissive).
pub mod aec;
pub mod align;
pub mod canary;
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
// converter (`vokra-convert::models::dnsmos`) landed 2026-08-03; this is the
// runtime binder shell (2026-08-05, loud-partial per RMVPE precedent — the
// CNN backbone op sequence is not primary-source-transcribable from the
// upstream `dnsmos_local.py` alone, only from the trained ONNX graph, so
// the forward returns `VokraError::UnsupportedOp` until the sidecar is
// extended with `vokra.dnsmos.{p808,p835}.topology` metadata and this
// module wires the CNN forward).
pub mod dnsmos_p808_p835;
// SoTA plan Phase 2 (2026-07-24): HuggingFace distil-whisper /
// distil-large-v3.5 — a distilled Whisper checkpoint that keeps the
// large-v3 encoder (32-layer / d_model=1280 / n_mels=128) intact and
// shrinks the decoder to 2 layers (same head width, same FFN dim, same
// large-v3 multilingual vocab at 51 866). No new op / kernel — the whole
// forward path is shared with the vanilla Whisper implementation, only
// `n_text_layer` differs. Config is transcribed verbatim from
// huggingface.co/distil-whisper/distil-large-v3.5/raw/main/config.json
// (CLAUDE.md ハルシネーション厳禁); real-checkpoint binding is a
// follow-up wave (T29-equivalent — delegates to `crate::whisper::WhisperModel`
// with the distil-shrunk decoder depth). Weights: MIT (Permissive — no
// runtime-side attribution obligation).
pub mod distil_whisper;
// SoTA plan Phase 5 JA-ASR-2 (2026-07-24): Kotoba Technologies
// **kotoba-whisper** — Whisper large-v3 encoder + a 2-layer decoder
// distilled on Japanese ReazonSpeech audio (multi-day Japanese ASR
// corpus). Same tensor topology as distil-large-v3.5 (identical shape
// quintuple `(1280, 32, 2, 128, 51866)`) but distinct upstream release
// (Kotoba Technologies) with **apache-2.0** weights (distil-whisper is
// MIT). Config is transcribed verbatim from
// huggingface.co/kotoba-tech/kotoba-whisper-v2.0/raw/main/config.json
// (CLAUDE.md ハルシネーション厳禁); real-checkpoint binding is a
// follow-up wave (T29-equivalent — delegates to `crate::whisper::WhisperModel`
// with the kotoba-shrunk decoder depth). Weights: Apache-2.0 (Permissive
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
// the "may need new op" note from the task); the shared primitive
// reused today is vokra_ops::ctc_decode.
pub mod omniasr_ctc;
// SoTA plan Phase 1-5 (2026-07-24): Zyphra Zonos-v0.1-transformer TTS
// (Apache 2.0). Single-stack GQA transformer with typed prefix conditioner
// (espeak / speaker / Fourier / integer) over DAC 44.1 kHz RVQ frames.
// Config is transcribed verbatim from huggingface.co/Zyphra/
// Zonos-v0.1-transformer/raw/main/config.json (CLAUDE.md ハルシネー
// ション厳禁); real-checkpoint binding is a follow-up wave (T29-equivalent).
pub(crate) mod mapped_weights;
pub mod mimi;
pub mod moshi;
// Microsoft DNS-Challenge NSNet2-baseline (arXiv:2005.07551, MIT — 2026-08-05
// runtime binder). Third denoise family member (alongside DFN3 =
// `vokra_ops::denoise` and RNNoise v0.2 = `rnnoise_v02`); a deliberately-
// weaker industry-baseline reference for quantization-CI cross-checks
// (CLAUDE.md audio dialect §"Speech Enhancement / AGC / AEC"). REAL forward:
// STFT (n_fft=512, hop=160, win=320, causal / non-center) → log-power
// feature → fc_in + 2×GRU + fc_1/fc_2/mask + sigmoid → gated STFT → streaming
// iSTFT. Reuses the tested `vokra_ops::rnnoise_gru_forward` primitive with
// an ONNX `[Z;R;H]` → rnnoise `[R;Z;H]` load-time permutation. Env-gated
// real-weight parity harness: `crates/vokra-models/tests/parity_nsnet2.rs`
// (VOKRA_NSNET2_REAL_GGUF + VOKRA_NSNET2_REAL_WAV).
pub mod nsnet2;
pub mod piper_plus;
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

// F0 (fundamental-frequency) extractor family (FR-OP-83). Skeleton — each
// member (`crepe`, and future PyIN / FCPE / Harvest / RMVPE siblings) exposes a
// GGUF `from_gguf` loader and an `extract` method whose real forward is a
// follow-up WP. Kept in its own block so `rustfmt`'s alphabetical sort inside
// consecutive `pub mod` blocks does not hijack the doc-preceded siblings above.
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
// Wave 2 2026-08-14 audit follow-up (vocoder recovery + music-und):
// vocos = standalone vocos runtime binder (loud-partial — ConvNeXt V2
// backbone missing from vokra-ops; iSTFT head available via
// `vokra_ops::istft`, Kokoro precedent).
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
// pending T5 relative_attention_bias primitive + MIDI event codec
// Rust port.
pub mod mt3;
pub mod vocos;
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
// primitives exist; BiLSTM extraction from silero_vad::model into
// vokra_ops::lstm is a follow-up. `facebook/demucs` MIT — first
// music-source-separation Permissive land after BS-Roformer
// Rejected).
pub mod demucs;
// Wave 5 2026-08-14 audit follow-up (music generation — first runtime
// binder): musicgen = Meta MusicGen family runtime binder (small 300M
// + medium 1.5B, CC-BY-NC-4.0 T4). Autoregressive transformer LM over
// EnCodec RVQ tokens (4 codebooks, 50 Hz frame rate, 32 kHz output)
// conditioned on frozen T5-base text encoder with delay pattern across
// codebooks. Real config / variant / from_gguf / weight-license
// surfacing; generate loud-partial pending T5 text-encoder forward +
// AR LM with 4-codebook delay-pattern + text-conditioned cross-
// attention (EnCodec RVQ decode primitive exists via
// vokra_ops::encodec_rvq_decode). Primary sources:
// huggingface.co/facebook/musicgen-{small,medium} +
// github.com/facebookresearch/audiocraft (MIT code) + arXiv:2306.05284
// (Copet et al. 2023). §3.1 row 399 = ☑ Research-only 2026-08-01
// yousan (X-Codec-2 T4 precedent). Loud-partial pattern per Wave 2-4
// precedent (vocos / bigvgan / snac / mt3 / kyutai_stt / parakeet_ctc
// / redimnet / sortformer).
pub mod musicgen;
// Wave 5 2026-08-14 audit follow-up (separation-runtime binder — LIB.RS
// RULE append at end with Wave 5 comment marker): Conv-TasNet
// (Luo & Mesgarani 2019, arXiv:1809.07454) — Convolutional Time-domain
// Speech Separation loud-partial runtime binder for the `conv_tasnet`
// converter arch. Real arch check + tensor-manifest non-emptiness gate
// + primary-source-transcribed ConvTasnetConfig (12-axis Asteroid
// Libri1Mix `enhsingle_16k` hold — the converter does NOT yet stamp
// `vokra.conv_tasnet.*` so a follow-up wave lands the strict axis read
// alongside the encoder-masker-decoder walk); separate() loud-partial
// pending Asteroid Python 1D Conv encoder + TCN masker + 1D
// ConvTranspose decoder composition
// (github.com/asteroid-team/asteroid). Speech-separation Copyleft
// (CC-BY-SA-4.0 T3 tier — §3.1 sign-off row `conv-tasnet-libri1mix`
// already ☑ Commercial by owner 2026-08-02 yousan) sibling to the
// music-source-separation `demucs` (MIT Permissive) landed just above.
pub mod conv_tasnet;
// Wave 5 2026-08-14 audit follow-up (separation-runtime binder — LIB.RS
// RULE append at end with Wave 5 comment marker): SpeechBrain SepFormer
// family (Subakan et al. 2021 / arXiv:2010.13154 §3, apache-2.0) — 7
// variants (wsj02mix / libri2mix / libri3mix / wham16k-enhancement /
// whamr16k / whamr8k / dns4-16k-enhancement) share the `sepformer`
// converter arch + this runtime binder. Real `from_gguf` (arch check +
// variant tag round-trip + n_out variant/stamp cross-check + non-empty
// tensor gate + weight-license class surfacing); `separate()` loud-
// partial pending dual-path Transformer masker composition (encoder +
// IntraTransformer + InterTransformer + `n_out`-way head + decoder)
// per `github.com/speechbrain/speechbrain/blob/develop/speechbrain/
// lobes/models/dual_path.py` + `resepformer.py` + arXiv:2010.13154.
// Sibling to the music-source-separation `demucs` (MIT Permissive) and
// the speech-source-separation `conv_tasnet` (CC-BY-SA-4.0 Copyleft T3)
// landed just above — §3.1 sign-off rows 364-370 all ☑ Commercial
// (2026-07-30 / 2026-08-01 yousan) per HF cardData apache-2.0.
pub mod sepformer;
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
// tokens (4 codebooks, 50 Hz, 32 kHz) conditioned on frozen T5-base text
// encoder with delay pattern across codebooks — MusicGen sibling by
// topology (Kreuk et al. 2023 arXiv:2209.15352), distinct by training
// corpus (environmental sounds / SFX vs music) and by arch tag
// `audiogen` (FR-EX-08 dispatch safety — audit follow-up 2026-08-14
// retags the converter from the Wave 5 shared `musicgen` arch tag so a
// future modality-specific head (SFX-only conditioning stack, stereo
// output head, per-class embedding table) does not silent-mis-bind
// against MusicGen's music-only runtime path). Real config / from_gguf
// / weight-license surfacing; generate loud-partial pending T5 text-
// encoder forward + AR LM with 4-codebook delay-pattern +
// text-conditioned cross-attention (EnCodec RVQ decode primitive exists
// via vokra_ops::encodec_rvq_decode — shared composition anchor with
// MusicGen; a single follow-up wave unblocks both binders because
// pieces (i) T5-base and (ii) AR-LM-with-delay-pattern are shared by
// construction). §3.1 row 402 = ☑ Research-only 2026-08-01 yousan
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
// Wave 8 2026-08-14 audit follow-up (LIB.RS RULE append at end with
// Wave 8 comment marker; **RETRY of Wave 7 silently-lost item** per
// WAVE 6/7 LESSON — write files to disk FIRST, then return
// IMPL_SCHEMA): emotion2vec+ Large (`emotion2vec/emotion2vec_plus_large`,
// MIT end-to-end) — 9-class speech emotion recognition self-supervised
// pretrain (Ma et al. 2024 ACL, arXiv:2312.15185). **First
// `category="emotion"` runtime binder in the converter tree**, sibling
// to the wav2vec2-SSL-lineage fleet (`wav2vec2_ctc` CTC ASR head /
// `wavlm_sv` XVector speaker head / `hubert` bare SSL / `data2vec-audio`
// masked-prediction encoder) — never silently shares arch since all
// four siblings expose completely different downstream heads on top of
// the shared wav2vec2 encoder lineage (FR-EX-08 forbids the silent
// shape misroute). Loud-partial pattern per panns / wavlm / storm /
// audioldm2 / musicgen / redimnet precedent — `from_gguf` REAL (strict
// arch check + non-empty tensor gate + weight-license class surfacing;
// converter does NOT stamp `vokra.emotion2vec.*` topology chunks so
// this binder mirrors the panns arch-only-gate posture, not the strict
// axis-array `wavlm_sv` posture) + `classify(pcm) -> Result<Vec<f32>>`
// UnsupportedOp naming (i) wav2vec2-style SSL Transformer encoder walk
// (base topology 12L/768H per wav2vec2 lineage, exact axes deferred to
// real-checkpoint dump), (ii) linear 9-way classifier head with all 9
// emotion labels echoed verbatim (Angry/Disgusted/Fearful/Happy/
// Neutral/Other/Sad/Surprised/`<unk>`) so a reader diagnosing the gap
// can cross-check `argmax` interpretation without walking the upstream
// `label.txt`, primary-source URLs `huggingface.co/emotion2vec/
// emotion2vec_plus_large` + `github.com/ddlBoJack/emotion2vec` +
// arXiv:2312.15185. §3.1 sign-off = Permissive (MIT, primary-source-
// verified by the converter 2026-07-25 — no owner action needed for
// MIT class per feedback-license-signoff-primary-source memory).
pub mod emotion2vec;
// Wave 8 2026-08-14 audit follow-up (LIB.RS RULE append at end with
// Wave 8 comment marker): Useful Sensors Moonshine ASR family
// (`UsefulSensors/moonshine-{tiny,base}`, MIT) — real-time
// transformer encoder-decoder ASR alternative to Whisper for edge
// (Jeffries et al. 2024, arXiv:2410.15608 "Moonshine: Speech
// Recognition for Live Transcription and Voice Commands"). **Distinct
// from every Whisper-family sibling** (whisper / distil_whisper /
// kotoba_whisper) in two significant ways: (1) **no mel front-end** —
// the model consumes raw 16 kHz PCM directly via a learned Conv1D
// stem (strides = [64, 3, 2] → 384x downsampling); (2) **RoPE + SwiGLU**
// activations rather than Whisper's sinusoidal + GELU. Loud-partial
// pattern per Wave 1-7 precedent (snac / wavlm / fsmn_vad /
// openwakeword / dnsmos_p808_p835 / storm / sepformer / demucs /
// conv_tasnet / musicgen / audiogen / audioldm2 / jasco / panns /
// llama_omni2 / emotion2vec): `from_gguf` real (arch check + variant
// discrimination via `vokra.model.name` + per-variant
// `MoonshineConfig` primary-source-transcribed hparams + weight-
// license class surfacing); `transcribe()` returns
// `VokraError::UnsupportedOp` naming the three exact missing pieces
// (i) raw-audio Conv1D stem walk, (ii) RoPE + SwiGLU transformer
// encoder-decoder forward, (iii) greedy / beam decoding +
// SentencePiece detokenize — every message cites all three primary
// source URLs (github.com/usefulsensors/moonshine +
// arXiv:2410.15608 + huggingface.co/UsefulSensors/moonshine-*) so a
// reader diagnosing the gap has exactly three anchors to walk.
// Consumes converter siblings `moonshine_tiny.rs` +
// `moonshine_base.rs` (both landed Wave 9 2026-08-02, §3.1 rows 421 +
// 422 both ☑ Commercial MIT by 2026-08-01 yousan — this runtime
// binder needs NO new §3.1 row). No new C ABI, no new Cargo.toml dep
// — cross-crate string handshake via duplicated
// `pub const ARCH = "moonshine"` (mirror of the converter's ARCH
// constant, preserving the layered convention `vokra-ops → nothing
// GGUF-aware`, `vokra-core → GGUF reader`, `vokra-models → GGUF
// binder`, `vokra-convert → GGUF writer`).
pub mod moonshine;
// Wave 8 2026-08-14 audit follow-up (LIB.RS RULE append at end with
// Wave 8 comment marker): Facebook Denoiser
// (`facebookresearch/denoiser`, **CC-BY-NC-4.0** — T4 tier
// research-only per docs/license-audit.md line 457 ☑ Research-only
// 2026-08-04 yousan sign-off, publish requires `--allow-noncommercial`)
// — real-time speech-enhancement waveform U-Net + LSTM (Defossez et al.
// 2020 arXiv:2006.12847 "Real Time Speech Enhancement in the Waveform
// Domain") runtime binder for the `facebook_denoiser` converter arch
// (Wave D T4, 2026-08-04, converter side already landed at
// `crates/vokra-convert/src/models/facebook_denoiser.rs`). Real
// `from_gguf` (arch check + non-empty tensor gate + weight-license
// class surfacing; converter does NOT stamp
// `vokra.facebook_denoiser.*` topology chunks — plain BF16 pass-through
// per NKF-AEC / RNNoise / NSNet2 GitHub-native precedent, so this
// binder mirrors the arch-only-gate posture rather than the strict
// axis-array `wavlm_sv` / `storm` posture); `denoise()` returns
// `UnsupportedOp` naming (i) 5-block time-domain waveform U-Net
// encoder (Conv1d(k=8, stride=4) + GLU stack, channel growth
// `H · 2^L`, causal denoiser H=48), (ii) 2-layer LSTM bottleneck
// (unidirectional for causal `denoiser_causal.th`, bidirectional for
// offline `master64.th`), (iii) 5-block symmetric transposed-conv
// decoder (`ConvTranspose1d(k=8, s=4)` + additive encoder-side skip
// connections BEFORE the transposed conv — NOT a HiFi-GAN upsampler
// which has no encoder-side skip and mel-input not waveform-input).
// Primary sources: `github.com/facebookresearch/denoiser` +
// arXiv:2006.12847. Distinct-arch discipline: sibling enhancement /
// denoise arches enumerated (`denoise` DFN3, `rnnoise`, `nsnet2`,
// `dnsmos`, `gtcrn`, `dtln_aec`, `mp_senet`, `frcrn`, `metricgan_plus`,
// `mossformer2_ss_16k`, `storm`, `sepformer`, `conv_tasnet`, `demucs`)
// — facebook-denoiser is the FIRST time-domain waveform U-Net + LSTM
// entry on the enhancement arm, sharing arch with any sibling would
// mis-route runtime dispatch (FR-EX-08). §3.1 sign-off already
// ☑ Research-only 2026-08-04 yousan (docs/license-audit.md line 457,
// Wave D T4 precedent, cc-by-nc-4.0 → NonCommercial T4 tier) — no
// additional license-audit action needed this wave (row already
// present).
pub mod facebook_denoiser;
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
// learnable 1-D encoder + DPRNN stack (which additionally needs a general
// recurrent primitive `vokra-ops` does not expose; the only recurrent kernels
// in the tree are the DFN3-specific GRU stack inside `vokra_ops::denoise`) +
// three transformer metric heads; the subjective head's `wav2vec2_base` SSL
// encoder (the shared wav2vec2-lineage gap with `emotion2vec` / `wavlm`) +
// attentive pool + linear projector + the NMR pairing — plus the upstream file
// to transcribe and the reserved `vokra.squim.{objective,subjective}.topology`
// chunk to stamp. No score value is ever fabricated (FR-EX-08).
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
// REAL: strict `vokra.model.arch` verification that refuses foreign GGUFs
// loudly with the `category = "eval"` sibling fleet enumerated (`dnsmos`
// P.808+P.835 CNN / `utmos` SaruLab wav2vec2 regression / `utmosv2` /
// `torchaudio_squim` STOI-PESQ-SI-SDR + MOS); manifest-derived variant
// discrimination (`pool_layers.` = upstream `class NISQA_DIM`, five cloned
// attention-pooling heads, vs `pool.` = upstream `class NISQA`, one head)
// with a per-clone presence gate so a missing head cannot silently shorten
// the score vector; framewise-CNN presence gate; non-empty tensor gate;
// the optional all-or-nothing `vokra.nisqa.*` front-end + topology groups
// including the upstream odd-`seg_length` parity requirement; and
// weight-license surfacing that fail-closes to `LicenseClass::Unknown`.
//
// LOUD-PARTIAL (CLAUDE.md 教訓 (a) 「loud-partial は fake-complete より
// honest」): `Nisqa::score` / `Nisqa::score_overall` return
// `VokraError::UnsupportedOp` naming three concrete blockers — (i) the
// MISSING PRIMITIVE `F.adaptive_max_pool2d`, which upstream `AdaptCNN`
// calls three times and which `vokra-ops` does not provide in any form
// (fixed-kernel pooling cannot reproduce it for a variable-length input);
// (ii) the MISSING METADATA, i.e. the three adaptive-pool output sizes and
// `td_sa_nhead`, which appear in NO weight tensor at all (`AdaptCNN` pool
// extents are pure config, and `nn.MultiheadAttention` packs every head
// into one `in_proj_weight`), plus the whole mel front-end; (iii) the
// MISSING SIDECAR `tools/parity/nisqa_v2_weight_prepare_checkpoint.py`,
// which the converter's docstring names but which has never been written.
// No fabricated MOS is ever emitted (FR-EX-08).
//
// LICENSING: the upstream README states verbatim that the CODE is MIT but
// that the released WEIGHTS (`nisqa.tar` / `nisqa_mos_only.tar` /
// `nisqa_tts.tar`) are CC-BY-NC-SA-4.0 →
// `LicenseClass::NonCommercialShareAlike` = **T4 / research-only**: never
// publishable without `publish-one.sh --allow-noncommercial`, and the
// share-alike obligation cascades to any derived GGUF. The converter's own
// `DEFAULT_LICENSE_SPDX = "cc-by-nc-sa-4.0"` handling was cross-checked
// against that primary source and is CORRECT — no discrepancy found.
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

// Wave B (2026-08-15) — TEN-VAD runtime binder (LIB.RS RULE: append at the
// END of the `pub mod` block with a Wave marker; do NOT alphabetize —
// rustfmt has reordered these before and broken a commit).
//
// Closes a real gap: `crates/vokra-convert/src/models/ten_vad.rs` (landed
// coverage-audit-2026-08-03 Wave A permissive continuation, 2026-08-04)
// produced a GGUF stamped `vokra.model.arch = "ten_vad"` that NOTHING in the
// workspace read back, so every converted checkpoint was unloadable. This
// module is that consumer.
//
// TEN-VAD (`github.com/TEN-framework/ten-vad`) is a compact (~306 KB ONNX
// bundle) real-time voice-activity detector positioned upstream as a
// ~5.5x-lighter alternative to Silero VAD v5. It is the THIRD first-class VAD
// topology under the shared `category = "vad-kws"` umbrella, and it exposes
// the same `vokra_core::engines::VadEngine` / `VadStreamHandle` seam as
// `silero_vad` and `fsmn_vad`, so a caller can swap between the three without
// rewriting call sites.
//
// REAL: strict `vokra.model.arch` verification that refuses a foreign GGUF
// loudly with the whole `vad-kws` sibling fleet enumerated (`silero-vad`
// 1:1-preserved pseudo-STFT + LSTM subgraph / `fsmn-vad` fbank + LFR + CMVN op
// stack / `openwakeword` + `openwakeword_op` keyword spotting); a tensor
// manifest walk with a non-empty gate AND a converter-contract dtype gate
// (`convert_ten_vad_file` has no quantization arm — it passes F32/F16/BF16
// through verbatim — so a K-quant tensor proves a foreign producer); a
// `require_tensor` name-resolution primitive that fails loudly NAMING the
// tensor and previewing the real manifest; a BF16 counter mirroring the
// converter's `TenVadReport::bf16_passthrough`; the OPTIONAL, all-or-nothing
// `vokra.ten_vad.*` topology group (absent -> `None`, half-stamped -> loud);
// a real hop-based streaming frame accumulator with a loud sample-rate gate
// (never a silent resample); and weight-license surfacing that fail-closes to
// `LicenseClass::Unknown`.
//
// LOUD-PARTIAL (CLAUDE.md 教訓 (a)「loud-partial は fake-complete より
// honest」): `TenVad::frame_probability` and the stream's `push_pcm` return
// `VokraError::UnsupportedOp` naming four concrete blockers — (i) the MISSING
// TENSOR-NAME MANIFEST, because the published artefact came from the generic
// `tools/parity/onnx_to_safetensors.py` bridge which passes ONNX initializer
// names through verbatim and nothing here records them; (ii) the MISSING
// TOPOLOGY AXES, since the converter stamps no `vokra.ten_vad.*` group at all,
// leaving hop size / feature width / hidden width / layer count unknown;
// (iii) the UNRESOLVED BACKBONE FAMILY, since the converter docstring says
// "small LSTM/GRU backbone" without committing to either and a coin flip here
// is SILENT-wrong (note `vokra_ops::rnnoise::gru_forward` is already
// shape-generic, so the blocker is the manifest and the axes, not the
// arithmetic); (iv) the MISSING LPCNet-DERIVED FRONT-END, since
// `vokra_ops::rnnoise::bark_filterbank` is RNNoise's FIXED 22-band table (same
// Xiph lineage, different hard-coded edges), not TEN-VAD's front-end. No
// fabricated speech probability is ever emitted (FR-EX-08). Deliberately NO
// `TenVadConfig::upstream_default()`: unlike `FsmnVadConfig`, TEN-VAD's axes
// are not stated in any available primary source, so a default would be
// invented numbers wearing an authoritative face (CLAUDE.md ハルシネーション厳禁).
//
// LICENSING: Apache-2.0 for the main project (`LicenseClass::Permissive`,
// mirroring the converter's `DEFAULT_LICENSE_SPDX`), but the LPCNet-derived
// DSP front-end bundled in the upstream distribution is separately
// BSD-3-Clause and requires NOTICE attribution for the LPCNet copyright when
// redistributing binaries that embed it — surfaced as the named constant
// `FRONTEND_LICENSE_SPDX` so the follow-up front-end-port wave has a greppable
// anchor for that obligation. `docs/license-audit.md` §3.1 sign-off stays
// BLANK (owner-only per `[[feedback-license-signoff-primary-source]]` — CC
// does NOT sign).
//
// Cross-crate string handshake via duplicated `pub const ARCH = "ten_vad"`
// (mirror of the converter's ARCH constant, preserving the layered convention
// `vokra-ops → nothing GGUF-aware`, `vokra-core → GGUF reader`,
// `vokra-models → GGUF binder`, `vokra-convert → GGUF writer`).
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
// (`pipecat-ai/smart-turn-v2`, BSD-2-Clause, w2v-BERT 2.0 backbone) answers
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
// REAL: strict `vokra.model.arch` verification refusing foreign GGUFs with
// the whole `category = "vad"` sibling fleet enumerated by the question
// each one actually answers (`fsmn-vad` / `firered_vad` / `silero-vad` /
// `pyannote-segmentation`, all per-frame) plus the bare backbone
// (`w2v-bert-2`, an SSL encoder with no turn head at all); a non-empty
// tensor gate; an encoder-stack presence gate; a classification-head
// presence gate that specifically stops a bare `facebook/w2v-bert-2.0`
// converted with the wrong `--model` flag from binding as a turn detector;
// the optional all-or-nothing `vokra.smart_turn.*` segment group; a
// validating `TurnPrediction` constructor (a NaN probability would compare
// false against every threshold, i.e. "still speaking" forever); and full
// loud input validation on `predict_endpoint` (zero / mismatched sample
// rate, empty segment, over-long segment) firing BEFORE the loud-partial
// gate so a malformed request always gets the specific diagnostic.
//
// LOUD-PARTIAL (CLAUDE.md 教訓 (a)「loud-partial は fake-complete より
// honest」): `SmartTurn::predict_endpoint` returns
// `VokraError::UnsupportedOp` naming four blockers — (i) the MISSING
// ADAPTER, a w2v-BERT-flavoured Conformer: `vokra_ops::conformer` DOES
// exist but is a NeMo port (Stacking subsampling stem, NeMo parameter
// layout, parakeet/canary consumers) while w2v-BERT 2.0 uses the HF
// `Wav2Vec2BertEncoder` variant that projects precomputed filterbank
// features, and the shapes are close enough that a substituted forward
// would RUN and return a plausible WRONG number rather than failing;
// (ii) the MISSING METADATA, since the converter is a verbatim float
// pass-through stamping no topology or front-end axes at all; (iii) the
// MISSING HEAD CONTRACT — one-logit-sigmoid vs two-logit-softmax and which
// index means "complete" are unrecoverable, and a guessed index has a 50%
// chance of INVERTING the decision; (iv) the MISSING SIDECAR
// `tools/parity/smart_turn_prepare_checkpoint.py`. No fabricated
// turn-completion probability is ever emitted (FR-EX-08) — a fake "turn
// complete" makes a realtime agent interrupt the user mid-sentence, so a
// plausible lie here is worse than a loud failure.
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
