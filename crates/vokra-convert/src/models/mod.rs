//! Per-model conversion routines (upstream checkpoint to GGUF builder).

// SoTA plan follow-on (2026-07-25): baichuan-inc/Baichuan-Audio
// (apache-2.0). Category: s2s. Baichuan Omni-1.5 = Whisper-Large enc +
// 8-layer RVQ 12.5Hz + Flow Matching mel + CosyVoice2 HiFi-GAN. Every
// F32 / F16 / BF16 tensor passes through verbatim following the
// qwen3_tts / vibevoice / voxcpm2 pattern; real-weight parity is
// deferred to owner (`docs/license-audit.md` §3.1 sign-off).
pub mod baichuan_audio;
pub mod bicodec;
pub(crate) mod campplus;
// SoTA plan Phase 2 (2026-07-24): NVIDIA Canary-1B-v2 — multilingual
// multi-task ASR / AST (25 European languages). FastConformer encoder
// (32 layers) + Transformer decoder (8 layers, AED). CC-BY 4.0 weight
// (AttributionRequired). Every F32 / F16 tensor passes through
// verbatim; every hparam on the model card is transcribed from it and
// every remaining axis from the shared FastConformer-Transformer AED
// reference config (fast-conformer_aed.yaml — the whole Canary family's
// reference). Reuses the `vokra_ops::conformer` (FastConformer encoder
// body via `Stacking { factor: 8 }`) and `vokra_ops::beam_search`
// (attention-decoder search) primitives — no per-model op duplication.
pub(crate) mod canary;
// SoTA plan Phase 3 (2026-07-24): Resemble AI Chatterbox-Multilingual TTS
// (MIT weight, Permissive) safetensors → GGUF with the `vokra.chatterbox.*`
// chunk group. Every F32 / F16 tensor passes through verbatim; every hparam
// is transcribed from the primary source (`github.com/resemble-ai/chatterbox`
// — `src/chatterbox/models/t3/`). No `config.json` on HF (the release stores
// hparams in Python code), so the converter takes no config side-car; the
// variant tag (multilingual vs english-only) is a caller argument, defaulted
// to Multilingual by [`convert`]. Reuses the shared HiFTChain seam
// (`vokra-models::cosyvoice2::hift_chain::HiFTChain`, SoTA plan §1(a) 訂正
// 2026-07-22) — no new op or backend kernel is added.
pub(crate) mod chatterbox;
// SoTA plan Phase 3 (2026-07-24): Resemble AI Chatterbox-Turbo TTS
// (MIT weight, Permissive) safetensors → GGUF with the
// `vokra.chatterbox_turbo.*` chunk group. Every F32 / F16 tensor
// passes through verbatim; every hparam is transcribed from the
// primary source `t3_turbo_v1.yaml`
// (`huggingface.co/ResembleAI/chatterbox-turbo`, fetched 2026-07-24).
// Distinct arch tag from base Chatterbox because Turbo swaps backbone
// family (gpt2-medium vs Llama_520M) + sample rate (32 kHz vs 24 kHz)
// + text vocabulary (50 276 vs 2454/704 — GPT-2 base 50 257 + 19
// paralinguistic tags from `added_tokens.json`); the terminal vocoder
// is still S3Gen HiFT-GAN so the shared HiFTChain seam
// (`vokra-models::cosyvoice2::hift_chain::HiFTChain`) applies —
// no new op or backend kernel is added.
pub(crate) mod chatterbox_turbo;
// SoTA plan Phase 3 (2026-07-24): Resemble AI Chatterbox-Nano TTS
// (MIT weight, Permissive) safetensors → GGUF with the
// `vokra.chatterbox_nano.*` chunk group. Every F32 / F16 tensor
// passes through verbatim; every hparam is transcribed from the
// primary source `t3_nano_v1.yaml`
// (`huggingface.co/ResembleAI/chatterbox-nano`, fetched 2026-07-24).
// Distinct arch tag from base Chatterbox + Turbo because Nano keeps
// base's Llama_520M backbone family but swaps sample rate (32 kHz vs
// 24 kHz) + text vocabulary (50 276 GPT-2 vs 2454/704) +
// stop-text sentinel (50 256 GPT-2 EOT vs 0); the terminal vocoder
// is still S3Gen HiFT-GAN so the shared HiFTChain seam
// (`vokra-models::cosyvoice2::hift_chain::HiFTChain`) applies —
// no new op or backend kernel is added.
pub(crate) mod chatterbox_nano;
pub(crate) mod cosyvoice2;
// SoTA plan Phase 3 (2026-07-24): FunAudioLLM Fun-CosyVoice3-0.5B (apache-2.0
// permissive) — same architecture as CosyVoice2 (Qwen2 LLM backbone +
// chunk-aware CFM + HiFTNet vocoder). The tensor walk / shape derivation /
// Q/K/V bias uniformity check is delegated to `cosyvoice2::convert_*`; this
// module rewrites the arch label, model name, provenance, and metadata chunk
// prefix so the runtime dispatches to `vokra-models::cosyvoice3` (Phase 3
// refinements DRSR + Core-Cocktail are training-side and leave the runtime
// operators byte-identical to CosyVoice2). No new op — a very cheap follow-on.
pub(crate) mod cosyvoice3;
pub(crate) mod csm;
pub(crate) mod dac;
// SoTA plan Phase 1-4 (2026-07-24): nari-labs Dia-1.6B (Apache 2.0)
// safetensors → GGUF with the `vokra.dia.*` chunk group. Every tensor passes
// through verbatim; every hparam is transcribed from the upstream config.json.
pub(crate) mod dia;
pub mod ecapa_tdnn;
pub mod freevc;
// SoTA plan Phase 2 (2026-07-24): HuggingFace distil-whisper /
// distil-large-v3.5 (MIT weight, Permissive) — a distilled Whisper checkpoint
// that keeps the large-v3 encoder intact (32 layers / d_model=1280 /
// n_mels=128) and shrinks the decoder to 2 layers. Every F32 / F16 tensor
// passes through verbatim under the upstream HF Whisper name; hparams ride
// the `vokra.whisper.*` chunk schema (schema shared with vanilla Whisper —
// the "very cheap follow-on" contract in the task). Reuses the shared
// Whisper op inventory (STFT / mel filterbank / GEMM / GEMV / softmax /
// layer-norm / GELU / conv1d) — no new op is added.
pub(crate) mod distil_whisper;
// SoTA plan Phase 5 JA-ASR-2 (2026-07-24): Kotoba Technologies
// **kotoba-whisper** family — Japanese-distilled Whisper (large-v3
// encoder + shrunk 2-layer decoder — same tensor topology as
// distil-large-v3.5, but distilled on ReazonSpeech Japanese audio
// and released under **apache-2.0**, distinct from distil-whisper's
// MIT). Every F32 / F16 tensor passes through verbatim under the
// upstream HF Whisper name; hparams ride the `vokra.whisper.*` chunk
// schema (schema shared with vanilla Whisper — the "very cheap
// follow-on" contract in the task). The JA-ASR-2 axis (data-driven
// decoder depth) is honored by the shape-driven `count_layers` walk
// — the converter reads `n_text_layer=2` from the checkpoint's
// tensor names, never hard-coding to 32. Reuses the shared Whisper
// op inventory (STFT / mel filterbank / GEMM / GEMV / softmax /
// layer-norm / GELU / conv1d) — no new op is added.
pub(crate) mod kotoba_whisper;
// M4-20 T12/T17: DeepFilterNet3 `denoise` → `vokra.denoise.*` GGUF (real
// checkpoint parse from the prepared safetensors, verbatim upstream names).
pub mod denoise;
pub mod funcodec;
pub mod kimi_audio;
pub mod knn_vc;
pub(crate) mod kokoro;
// SoTA plan Phase 2 (2026-07-24): Kyutai STT-2.6B-EN (CC-BY 4.0 weight,
// AttributionRequired) safetensors → GGUF with the `vokra.kyutai_stt.*`
// chunk group. Every F32 / F16 tensor passes through verbatim; every
// hparam is transcribed from the upstream config.json. The upstream
// release is BF16 and the streaming-BF16 pass-through path is a follow-up
// (T29-equivalent — the Moshi pattern).
pub(crate) mod kyutai_stt;
pub mod meanvc;
pub(crate) mod mimi;
pub(crate) mod moshi;
pub mod neucodec;
// SoTA plan Phase 2 (2026-07-24): NVIDIA Parakeet-TDT-0.6B-v3 — English
// ASR (FastConformer encoder + TDT decoder). CC-BY 4.0 weight
// (AttributionRequired). Every F32 / F16 tensor passes through
// verbatim; every hparam is transcribed from the upstream
// `config.json` (encoder_config + decoder + joint / TDT). Reuses the
// `vokra_ops::conformer` + `vokra_ops::rnnt_decode` primitives — no
// per-model op duplication.
pub(crate) mod parakeet;
// SoTA plan Phase 2 (2026-07-24): NVIDIA Parakeet-CTC-1.1B — English ASR
// (FastConformer encoder + CTC head, no RNN-T prediction network). CC-BY
// 4.0 weight (AttributionRequired). Every F32 / F16 tensor passes through
// verbatim; every hparam is transcribed from the upstream `config.json`
// (encoder_config + top-level vocab_size / pad_token_id — no decoder or
// joint section exists for CTC). Reuses the `vokra_ops::conformer` +
// `vokra_ops::ctc_decode` primitives — no per-model op duplication.
pub(crate) mod parakeet_ctc;
// SoTA plan Phase 2 (2026-07-24): Meta omniASR-CTC-1B — 1600+ language
// multilingual ASR (wav2vec 2.0 waveform-in encoder + single-Linear CTC
// head, no RNN-T prediction network). Apache-2.0 weight (Permissive).
// Every F32 / F16 tensor passes through verbatim; every hparam is
// transcribed from the upstream fairseq2 registry walk
// (`omnilingual_asr/models/wav2vec2_asr/config.py::_1b_asr` →
// `wav2vec2_ssl/config.py::_1b_ssl` →
// `fairseq2/models/wav2vec2/config.py::large_lv60k`) — the HF release
// carries no `config.json`. Reuses the `vokra_ops::ctc_decode`
// primitive; the wav2vec 2.0 encoder body is a distinct topology
// from FastConformer (no shared `vokra_ops::wav2vec2_encoder` op today —
// deliberate "may need new op" follow-up).
pub(crate) mod omniasr_ctc;
pub mod openvoice_v2;
pub(crate) mod piper_plus;
// SoTA plan Phase 3 (2026-07-24): Alibaba **Qwen3-TTS-12Hz-0.6B-Base**
// (apache-2.0 end-to-end weight) safetensors → GGUF with the
// `vokra.qwen3_tts.*` chunk group. Discrete multi-codebook LM
// (Qwen3-flavour 28-layer talker + 5-layer parallel code predictor +
// shared Qwen3-TTS-Codec seam via `vokra_ops::qwen3_tts_codec`).
// Every F32 / F16 tensor passes through verbatim; every hparam is
// transcribed from the primary source
// `huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base/raw/main/config.json`
// (talker.* + code_predictor.*) plus README.md (speaker encoder
// 24 kHz / 1024-dim). Distinct arch tag from CosyVoice2/3 because
// Qwen3-TTS is codec-LM not vocoder-LM — the terminal step is
// qwen3_tts_codec, NOT HiFTChain.
pub(crate) mod qwen3_tts;
pub(crate) mod silero;
pub mod speechtokenizer;
// SoTA plan Phase 3 (2026-07-25): StepFun **Step-Audio-2-mini**
// (apache-2.0 end-to-end weight) safetensors → GGUF skeleton. 8B S2S
// with a dual codebook (semantic 1024 + acoustic 4096) and a
// flow-matching mel decoder. This is the pass-through skeleton
// (`convert_step_audio2_mini_file`) — every F32 / F16 / BF16 tensor
// passes through verbatim; real-weight parity is deferred to owner
// (docs/license-audit.md §3.1 sign-off). Distinct arch tag from every
// sibling — silently sharing would mis-route the runtime dispatch.
pub mod step_audio2_mini;
pub(crate) mod utmos;
// SoTA plan Phase 4 (2026-07-24): OpenBMB **VoxCPM-0.5B** (apache-2.0
// end-to-end weight) safetensors → GGUF with the `vokra.voxcpm2.*` and
// `vokra.vae_continuous.*` chunk groups. NEW class of TTS vs every
// earlier target — the terminal decoding hop is a continuous VAE
// decoder consuming flow-matching sampler output (not vocoder-LM
// HiFTChain, not codec-LM RVQ / FSQ). Topology: MiniCPM-4 LM backbone
// (24-layer / 1024d / GQA 16 Q ÷ 2 KV / SwiGLU 4096 / RoPE θ=10000
// with longrope scaling / RMSNorm ε=1e-5 / vocab=73448) + 6-layer
// residual acoustic LM + 4-layer local encoder + 4-layer local DiT +
// UnifiedCFM flow-matching sampler (Euler / inference_cfg_rate=2.0) +
// AudioVAE V2 continuous encoder / decoder (16 kHz PCM in → 25 Hz
// latents → 48 kHz PCM out) + inline scalar-quantization bottleneck
// (`scalar_quantization_latent_dim=256`, `scalar_quantization_scale=9`).
// Every F32 / F16 tensor passes through verbatim; every hparam is
// transcribed from the primary source
// `huggingface.co/openbmb/VoxCPM-0.5B/raw/main/config.json` +
// `openbmb/VoxCPM/src/voxcpm/modules/audiovae/audio_vae_v2.py`
// (`AudioVAEConfig` defaults). Distinct arch tag from CosyVoice2/3 /
// Qwen3-TTS / Chatterbox family because VoxCPM's terminal step is
// vae_continuous_decode, NOT HiFTChain or any RVQ / FSQ codec.
pub(crate) mod voxcpm2;
// SoTA plan Phase 4 (2026-07-24): Microsoft **VibeVoice-1.5B** (MIT
// end-to-end weight) safetensors → GGUF with the `vokra.vibevoice.*`
// chunk group. SECOND consumer of the continuous VAE + diffusion
// decoder class (after VoxCPM-0.5B) — but where VoxCPM uses a
// UnifiedCFM flow-matching sampler, VibeVoice uses a **DDPM** sampler
// (v-prediction + cosine β schedule). Topology: Qwen2 decoder LM
// (28-layer / 1536d / MHA n_head=12 n_head_kv=2 (GQA ratio 6) /
// SwiGLU 8960 / RoPE θ=1_000_000 / RMSNorm ε=1e-6 / vocab=151_936 /
// max_positions=65_536 / tie_word_embeddings=true) + acoustic σ-VAE
// tokenizer (vae_dim=64, mirror-symmetric encoder/decoder at 24 kHz,
// encoder_ratios=[8,5,5,4,2,2] → 7.5 Hz frame rate) + semantic
// tokenizer (encoder-only deterministic, vae_dim=128) + 4-layer
// AdaLN-modulated MLP diffusion head (hidden=1536, head_ffn_ratio=3.0 →
// ffn_dim=4608, latent_size=64, prediction_type="v_prediction",
// diffusion_type="ddpm", ddpm_num_steps=1000,
// ddpm_num_inference_steps=20, ddpm_beta_schedule="cosine"). Every
// F32 / F16 tensor passes through verbatim; every hparam is
// transcribed from the primary sources
// `huggingface.co/microsoft/VibeVoice-1.5B/raw/main/config.json` and
// `github.com/microsoft/VibeVoice/blob/main/vibevoice/modular/
// configuration_vibevoice.py`. Distinct arch tag from VoxCPM /
// CosyVoice2/3 / Qwen3-TTS / Chatterbox family — silently sharing
// would misroute the runtime dispatch (VoxCPM → flow_sample,
// VibeVoice → ddpm_sample).
pub(crate) mod vibevoice;
// SoTA plan Phase 5 JA-TTS-1 (2026-07-24): Aratako **Irodori-TTS-500M-v3**
// Japanese TTS (MIT weight + code, verified via
// `gh api /repos/Aratako/Irodori-TTS/license` → `MIT`, fetched
// 2026-07-24 — CLAUDE.md「ハルシネーション厳禁」) safetensors → GGUF
// with the `vokra.irodori.*` chunk group. A Rectified-Flow Diffusion
// Transformer (RF-DiT) over the paired `Semantic-DACVAE-Japanese-32dim`
// codec (32-d continuous latent → 48 kHz PCM). Every F32 / F16 tensor
// passes through verbatim; every hparam is transcribed from
// `train_500m_v3_phase1_body.yaml` + `train_500m_v3_phase2_duration.yaml`
// + `irodori_tts/config.py::ModelConfig` at
// `github.com/Aratako/Irodori-TTS`. Distinct arch tag from every
// sibling — silently sharing would misroute the runtime dispatch
// (VibeVoice → ddpm_sample, VoxCPM → EpsS flow_sample, Irodori →
// Linear/Sway flow_sample with a distinct latent width 32 vs the
// Phase-4 siblings' 64). No side-car config today: every field is
// fixed for the 500M-v3 release and byte-parallel to the transcribed
// constants; a future 600M VoiceDesign / 2.5B variant that reshapes
// the DiT or adds caption conditioning would demand `--config`. No
// new op or backend kernel — reuses the shared
// `vokra_ops::flow_sampler` primitive (M3-05, `OdeSolver::Euler` +
// `Schedule::Linear` | `Schedule::Sway`) and the shared
// `crate::codec::DacCodecGguf` seam for the paired DACVAE decode.
pub(crate) mod irodori;
// SoTA plan Phase 5 JA-TTS-2 (2026-07-24): ESPnet-family Japanese
// **plain VITS** (Kim et al. 2021 VITS + HiFi-GAN generator) —
// architecture Apache-2.0 (ESPnet `espnet2/gan_tts/vits/`) + MIT
// (`jaywalnut310/vits` reference). Distinct arch tag from piper-plus
// because plain VITS decodes through a HiFi-GAN generator directly
// while piper-plus (MB-iSTFT-VITS2) decodes through a sub-band iSTFT
// + PQMF post-net (silently sharing an arch tag would misroute the
// runtime dispatch). Every F32 / F16 tensor passes through verbatim;
// every hparam is transcribed from the primary sources
// `egs2/jsut/tts1/conf/tuning/train_vits.yaml` +
// `egs2/jvs/tts1/conf/tuning/finetune_vits.yaml` +
// `espnet2/gan_tts/vits/{vits,generator}.py` (fetched 2026-07-24 —
// CLAUDE.md「ハルシネーション厳禁」). Reuses the shared
// `vokra_ops::hifigan_generator` (M3-07, FR-OP-10) primitive via
// `VitsJaConfig::to_hifigan_attrs`; no new op or backend kernel is
// added. **⚠️  Weight redistribution default**: the JSUT / JVS /
// COEIROINK corpus terms forbid trained-weight redistribution, so
// the provenance stamp defaults to
// `LicenseClass::RedistributionForbidden`. A user who trained on a
// permissive corpus overrides at the outer
// `convert_file --license <spdx>` boundary.
pub(crate) mod vits_ja;
pub(crate) mod voxtral;
pub(crate) mod whisper;
// SoTA plan Phase 5 codec (2026-07-25): fnlp XY_Tokenizer_TTSD_V0
// (apache-2.0) safetensors → GGUF. 1 kbps RVQ-8 @ 12.5 Hz — the codec
// half of MOSS-TTSD. F32 / F16 / BF16 pass-through following the
// qwen3_tts / vibevoice / voxcpm2 landed contract.
pub mod xy_tokenizer;
// SoTA plan Phase 1-5 (2026-07-24): Zyphra Zonos-v0.1-transformer
// (Apache 2.0) safetensors → GGUF with the `vokra.zonos.*` chunk group.
// Every float tensor passes through verbatim; every hparam (including the
// 7 typed prefix conditioners) is transcribed from the upstream config.json.
pub(crate) mod zonos;
