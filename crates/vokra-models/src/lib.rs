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
