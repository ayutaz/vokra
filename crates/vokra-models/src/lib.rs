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
pub mod canary;
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
pub mod silero_vad;
pub mod speaker;
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
pub mod voxtral;
pub mod whisper;
pub mod zonos;

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
